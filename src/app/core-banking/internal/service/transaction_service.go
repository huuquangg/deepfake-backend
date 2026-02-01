package service

import (
	"core-banking/internal/dto"
	"core-banking/internal/models"
	"core-banking/internal/repository"
	"errors"

	"github.com/google/uuid"
)

type TransactionService struct {
transactionRepo *repository.TransactionRepository
accountRepo     *repository.AccountRepository
}

func NewTransactionService(transactionRepo *repository.TransactionRepository, accountRepo *repository.AccountRepository) *TransactionService {
return &TransactionService{
transactionRepo: transactionRepo,
accountRepo:     accountRepo,
    }
}

// Transfer performs money transfer between accounts
func (s *TransactionService) Transfer(request *models.TransactionRequest, userID int) (*dto.TransactionResponse, error) {
// 1. Get sender's account
fromAccount, err := s.accountRepo.GetAccountByUserID(userID)
if err != nil {
return nil, errors.New("sender account not found")
    }

// 2. Get receiver's account
toAccount, err := s.accountRepo.GetAccountByNumber(request.ToAccountNumber)
if err != nil {
return nil, errors.New("receiver account not found")
    }

// 3. Validate transfer
if err := s.ValidateTransfer(fromAccount, toAccount, request.Amount); err != nil {
return nil, err
    }

// 4. TODO: Deepfake detection with FaceImageBase64
// Will implement later when connecting to AI service

// 5. Perform transfer (update balances)
newFromBalance := fromAccount.Balance - request.Amount
newToBalance := toAccount.Balance + request.Amount

// Update sender balance
if err := s.accountRepo.UpdateBalance(fromAccount.ID, newFromBalance); err != nil {
return nil, errors.New("failed to update sender balance")
    }

// Update receiver balance
if err := s.accountRepo.UpdateBalance(toAccount.ID, newToBalance); err != nil {
// Rollback sender balance if receiver update fails
s.accountRepo.UpdateBalance(fromAccount.ID, fromAccount.Balance)
return nil, errors.New("failed to update receiver balance")
    }

// 6. Create transaction record (THÊM to_account_id)
transaction := &models.Transaction{
TransactionID:   uuid.New().String(),
FromAccountID:   fromAccount.ID,
ToAccountNumber: request.ToAccountNumber,
ToAccountID:     &toAccount.ID, // LƯU ID người nhận
Amount:          request.Amount,
Description:     request.Description,
Status:          "COMPLETED",
    }

if err := s.transactionRepo.CreateTransaction(transaction); err != nil {
return nil, errors.New("failed to create transaction record")
    }

// 7. Return response
return &dto.TransactionResponse{
Transaction: dto.ToTransactionDTO(transaction),
Message:     "Transfer successful",
Success:     true,
    }, nil
}

// ValidateTransfer validates transfer conditions
func (s *TransactionService) ValidateTransfer(fromAccount, toAccount *models.Account, amount float64) error {
// Check if amount is positive
if amount <= 0 {
return errors.New("amount must be greater than 0")
    }

// Check if sender has sufficient balance
if fromAccount.Balance < amount {
return errors.New("insufficient balance")
    }

// Check if not transferring to self
if fromAccount.AccountNumber == toAccount.AccountNumber {
return errors.New("cannot transfer to your own account")
    }

// Check if accounts are active
if fromAccount.Status != "ACTIVE" {
return errors.New("sender account is not active")
    }
if toAccount.Status != "ACTIVE" {
return errors.New("receiver account is not active")
    }

return nil
}

// GetTransactionHistory gets transaction history for a user (cho GỬI VÀ NHẬN)
func (s *TransactionService) GetTransactionHistory(userID int) ([]*dto.TransactionDTO, error) {
// Get user's account
account, err := s.accountRepo.GetAccountByUserID(userID)
if err != nil {
return nil, errors.New("account not found")
    }

//   Get transactions WHERE user is SENDER OR RECEIVER
transactions, err := s.transactionRepo.GetTransactionsByAccountID(account.ID)
if err != nil {
return nil, err
    }

return dto.ToTransactionDTOList(transactions), nil
}